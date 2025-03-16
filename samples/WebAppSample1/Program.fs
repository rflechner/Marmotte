namespace WebAppSample1
#nowarn "20"
open Microsoft.AspNetCore.Builder
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Hosting
open Scalar.AspNetCore

module Program =
    let exitCode = 0

    [<EntryPoint>]
    let main args =

        let builder = WebApplication.CreateBuilder(args)

        builder.Services.AddAuthorization()
        
        builder.Services.AddEndpointsApiExplorer()
        
        builder.Services.AddOpenApi Marmotte.OpenApiExtensions.configureOpenApi

        let app = builder.Build()

        if app.Environment.IsDevelopment()
        then
            app.MapOpenApi() |> ignore
            app.MapScalarApiReference() |> ignore
        
        app.UseHttpsRedirection()

        app.UseAuthorization()
        app.UseRouting()

        
        app |> RestApp.run

        exitCode