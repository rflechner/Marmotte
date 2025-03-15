namespace WebAppSample1
#nowarn "20"
open System
open System.Threading.Tasks
open Marmotte.RestfulDsl
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Http
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Hosting
open Scalar.AspNetCore
open Marmotte

// https://learn.microsoft.com/fr-fr/aspnet/core/fundamentals/openapi/aspnetcore-openapi?view=aspnetcore-9.0&tabs=visual-studio
// https://learn.microsoft.com/en-us/aspnet/core/fundamentals/openapi/using-openapi-documents?view=aspnetcore-9.0#use-scalar-for-interactive-api-documentation

module Program =
    let exitCode = 0

    type UserDto = { Id: int; Name: string }
    
    let getUserById (id: int) : Task<UserDto option> =
        // Simule l'accès à une base de données ou autre stockage
        Task.FromResult(
            if id = 42 then Some { Id = 42; Name = "Douglas Adams" } else None
        )

    let getUserHandler (ctx: HttpContext) (_: RequestDelegate) =
        task {
            match ctx.Request.RouteValues.TryGetValue("id") with
            | true, (:? string as idStr) when idStr |> Int32.TryParse |> fst ->
                let id = int idStr
                let! user = getUserById id
                match user with
                | Some u ->
                    ctx.Response.StatusCode <- 200
                    ctx.Response.ContentType <- "application/json"
                    do! ctx.Response.WriteAsJsonAsync(u)
                | None ->
                    ctx.Response.StatusCode <- 404
                    do! ctx.Response.WriteAsync("User not found.")
            | _ ->
                ctx.Response.StatusCode <- 400
                do! ctx.Response.WriteAsync("Invalid ID provided.")
        }
    
    [<CLIMutable>]
    type CustomerRequestModel = { Id: int }
    
    [<CLIMutable>]
    type SearchCustomerRequestModel = { NamePattern: string }
    
    type Customer = { Id: int; Name: string; Birthday: DateOnly }
    
    [<CLIMutable>]
    type UserRequestModel = { Id: int }

    [<EntryPoint>]
    let main args =

        let builder = WebApplication.CreateBuilder(args)

        builder.Services.AddAuthorization()
        
        builder.Services.AddEndpointsApiExplorer()
        builder.Services.AddOpenApi()

        let app = builder.Build()

        if app.Environment.IsDevelopment()
        then
            app.MapOpenApi() |> ignore
            app.MapScalarApiReference() |> ignore
            //app.UseOpenApiExplorer() |> ignore
        
        app.UseHttpsRedirection()

        app.UseAuthorization()
        app.UseRouting()

        
        let getUser (ctx: HttpContext) =
            task {
                match ctx.Request.RouteValues.TryGetValue "id" with
                | true, (:? string as idStr) when idStr |> Int32.TryParse |> fst ->
                    let id = int idStr
                    let! user = getUserById id
                    match user with
                    | Some u -> return! Results.Json(u).ExecuteAsync(ctx)
                    | None -> return! Results.NotFound("User not found").ExecuteAsync(ctx)
                | _ -> return! Results.BadRequest("Invalid ID").ExecuteAsync(ctx)
            }
        
        app |> Http.get "/users/{id:int}" getUser
        
        let getCustomer (_: HttpContext) (request: CustomerRequestModel) =
            task {
                printfn "request: %A" request
                match request.Id with
                | 42 ->
                    let birthDay = DateTime.Today.AddYears(-30) |> DateOnly.FromDateTime
                    let name = $"Customer {request.Id}"
                    let customer : Customer = { Id=request.Id; Name=name; Birthday=birthDay }
                    return Result.Ok customer
                | _ -> return Result.Error "Customer not found"
            }
        
        let searchCustomer (_: HttpContext) (request: SearchCustomerRequestModel) =
            task {
                printfn "request: %A" request
                match request.NamePattern with
                | "joe*" ->
                    let birthDay = DateTime.Today.AddYears(-50) |> DateOnly.FromDateTime
                    let customer : Customer = { Id=48; Name="Joe Resta"; Birthday=birthDay }
                    return Result.Ok customer
                | _ -> return Result.Error "Customer not found"
            }

        // app |> Restful.addResource "Customer" (
        //         fun ctx ->
        //             ctx |> Restful.get "/{id:int}" getCustomer id
        //             ctx |> Restful.post "/search" searchCustomer (fun c -> { c with OperationName=Some "SearchCustomer" })
        //         )

        restful app {
            resource "Customer"
            get "/{id:int}" getCustomer
            post "/search" searchCustomer
            
            resource "User"
            
            get "/{id:int}" (
                    fun (_: HttpContext) (request:UserRequestModel) ->
                        task {
                            match request.Id with
                            | 42 ->
                                let name = $"User {request.Id}"
                                let user : UserDto = { Id=request.Id; Name=name }
                                return Result.Ok user
                            | _ -> return Result.Error "User not found"
                        }
                    )
        }
        
        app.Run()

        exitCode