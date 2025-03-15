module WebAppSample1.RestApp

open System
open System.Threading.Tasks
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Http
open Marmotte
open Marmotte.RestfulDsl
open WebAppSample1.Dto

let getUserById (id: int) : Task<UserDto option> =
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

// this example demonstrates how to use classic IResult and manually read requests
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

// this example demonstrates how to use Restful utilities
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
    
let run (app: WebApplication) =
    
    // this example demonstrates how to use classic IResult and manually read requests
    app |> Http.get "/users/{id:int}" getUser |> ignore
    
    // If you don't want to use the computation expression, you can uncomment following code

    // app |> Restful.addResource "Customer" (
    //         fun ctx ->
    //             ctx |> Restful.get "/{id:int}" getCustomer id
    //             ctx |> Restful.post "/search" searchCustomer (fun c -> { c with OperationName=Some "SearchCustomer" })
    //         )
    // app.Run()

    // computation expression creates routes
    // this example demonstrates how to use Restful utilities
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
    } |> _.Run()
    
    